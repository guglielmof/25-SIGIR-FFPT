fullM = readtable("tmp_perf.csv");

measure = string(unique(fullM.measure));
collection = string(unique(fullM.collection));
user_model = string(unique(fullM.user_model));
encoder = string(unique(fullM.encoder));

C = cell(1,length(measure)*length(collection)*length(user_model)*length(encoder));

ic = 0;
for i=1:length(measure)
    disp(measure{i})
    for j=1:length(collection)
        for k=1:length(user_model)
            for h=1:length(encoder)
                ic = ic + 1;
                localM = fullM(fullM.measure == measure(i) & ...
                               fullM.collection == collection(j) & ...
                               fullM.user_model == user_model(k) & ...
                               fullM.encoder == encoder(h) ...
                    , ["query_id", "model", "value"]);
                tblstats2 = grpstats(localM,"model","mean","DataVars",["value"]);
                means = tblstats2(:, "mean_value");
                means.best = zeros(height(means), 1);
                means.tpgr = zeros(height(means), 1);

                [val, idx] = max(means{:, "mean_value"});
                means{idx, "best"} = 1;
                bm = means.Properties.RowNames{idx};
                
                g = table2cell(localM(:, ["query_id", "model"]));
                y = localM{:,3}.';
                [~,~,stats] = anovan(y,{localM{:,1}.' localM{:,2}.'},  "Varnames",["query_id","model"], 'display', 'off');
                [c,~,~,gnames] = multcompare(stats , 'Display','off', "Dimension", 2);
                gnames = replace(gnames,"model=","");
                

                group_index = find(strcmp(gnames, bm));
                equiv = c((c(:, 1)==group_index|c(:, 2)==group_index) & (c(:, 6)>0.05), 1:2);
                equiv = unique([equiv(:, 1); equiv(:, 2)]);
                equiv = gnames(equiv);
                means{equiv, "tpgr"} = 1;


                means.measure = repelem(measure(i), height(means)).';
                means.collection = repelem(collection(j), height(means)).';
                means.user_model = repelem(user_model(k), height(means)).';
                means.encoder = repelem(encoder(h), height(means)).';
                means.model = string(means.Properties.RowNames);
                means.Properties.RowNames = {};
                C{ic} = means;

            end
        end
    end
end
out = vertcat(C{:});
writetable(out,"out.csv");

%M = fullM(fullM.user_model == "perfect", ["query_id", "model", "value"]);
%g = table2cell(M(:, ["query_id", "model"]));
%y = M{:,3}.';
%[~,~,stats] = anovan(y,{M{:,1}.' M{:,2}.'},  "Varnames",["query_id","model"], 'display', 'off');
%[c,~,~,gnames] = multcompare(stats , 'Display','off', "Dimension", 2);